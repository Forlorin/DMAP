import json
import re
from dataclasses import dataclass
from PIL import Image
import os
import pymupdf  # type: ignore[import-untyped]
from tqdm import tqdm
from datetime import datetime
import glob
from dotmap import DotMap  # type: ignore[import-untyped]
import roman  # type: ignore[import-untyped]
from typing import Literal, Optional


@dataclass
class PageContent:
    page: int
    image: Image.Image | None
    image_path: str
    text: str


@dataclass
class Figure:
    page: int
    image_path: str
    text: str


class BaseDataset:
    def __init__(self, config, dataset_name):
        cfg = DotMap(config)
        self.dataset_name = dataset_name
        cfg.dataset.name = dataset_name
        self.data_dir = cfg.dataset.data_dir.format(**cfg)
        cfg.dataset.data_dir = self.data_dir
        self.result_dir = cfg.dataset.result_dir.format(**cfg)
        self.extract_path = cfg.dataset.extract_path.format(**cfg)
        self.document_path = cfg.dataset.document_path.format(**cfg)
        self.sample_path = cfg.dataset.sample_path.format(**cfg)
        self.sample_with_retrieval_path = cfg.dataset.sample_with_retrieval_path.format(
            **cfg
        )
        self.question_key = cfg.dataset.question_key
        self.use_mix = cfg.dataset.use_mix
        self.r_text_key = cfg.retrieval.r_text_key.format(**cfg)
        self.r_image_key = cfg.retrieval.r_image_key.format(**cfg)
        self.top_k = cfg.dataset.top_k
        self.page_id_key = cfg.dataset.page_id_key
        self.max_page = cfg.dataset.max_page
        self.max_character_per_page = cfg.dataset.max_character_per_page
        self.pdffigure2_extract_path = cfg.dataset.pdffigure2_extract_path.format(**cfg)
        self.pdffigure2_path = cfg.dataset.pdffigure2_path.format(**cfg)
        self.summary_path = cfg.dataset.summary_path.format(**cfg)
        self.index_path = cfg.dataset.index_path.format(**cfg)
        self.sample_select_num = cfg.run_args.sample_select_num
        self.sqlite_path = cfg.dataset.sqlite_path.format(**cfg)
        self.IM_FILE = (
            lambda doc_name, index: f"{self.extract_path}/{doc_name}_{index}.png"
        )
        self.TEXT_FILE = (
            lambda doc_name, index: f"{self.extract_path}/{doc_name}_{index}.txt"
        )
        self.EXTRACT_DOCUMENT_ID = lambda sample: re.sub(
            "\\.pdf$", "", sample["doc_id"]
        ).split("/")[-1]
        self.SUMMARY_FILE = (
            lambda doc_name: f"{self.summary_path}/{doc_name}_summary.md"
        )
        self.INDEX_FILE = lambda doc_name: f"{self.index_path}/{doc_name}_index.md"
        current_time = datetime.now()
        self.time = current_time.strftime("%Y-%m-%d-%H-%M")

    def load_data(self, use_retrieval=True):
        """从路径中加载文档 通过读取对象本身的config参数，得到数据集中的所有数据的索引，文件格式为json

        Args:
            use_retrieval (bool, optional): _description_. Defaults to True.

        Returns:
            _type_:
        """
        path = self.sample_path
        if use_retrieval:
            try:
                assert os.path.exists(self.sample_with_retrieval_path)
                path = self.sample_with_retrieval_path
            except:
                print("Use original sample path!")
        print(f"dataset path:{path}")
        assert os.path.exists(path)
        with open(path, "r") as f:
            samples = json.load(f)
        if self.sample_select_num != -1:
            samples = samples[0 : self.sample_select_num]
        return samples

    def load_batch_data(self, start: int, end: int, use_retrieval=True):
        """从路径中加载文档 通过读取对象本身的config参数，得到数据集中的所有数据的索引，文件格式为json

        Args:
            use_retrieval (bool, optional): _description_. Defaults to True.

        Returns:
            _type_:
        """
        path = self.sample_path
        if use_retrieval:
            try:
                assert os.path.exists(self.sample_with_retrieval_path)
                path = self.sample_with_retrieval_path
            except:
                print("Use original sample path!")
        print(f"dataset path:{path}")
        assert os.path.exists(path)
        with open(path, "r") as f:
            samples = json.load(f)
        samples = samples[start:end]
        return samples

    def dump_data(self, samples, use_retrieval=True):
        if use_retrieval:
            path = self.sample_with_retrieval_path
        else:
            path = self.sample_path

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(samples, f, indent=4)
        return path

    def load_latest_results(self):
        print(self.result_dir)
        path = find_latest_json(self.result_dir)
        assert isinstance(path, str)
        with open(path, "r") as f:
            samples = json.load(f)
        if self.sample_select_num != -1:
            samples = samples[0 : self.sample_select_num]
        return samples, path

    def dump_results(self, samples):
        os.makedirs(self.result_dir, exist_ok=True)
        path = os.path.join(self.result_dir, self.time + ".json")
        print(f"save_to {path}")
        with open(path, "w") as f:
            json.dump(samples, f, indent=4)
        return path

    def get_sample_question(self, sample) -> str:
        return sample[self.question_key]

    def load_sample_retrieval_data(
        self, sample, extra_contents: list[PageContent] = []
    ):
        content_list = self.load_processed_content(sample, disable_load_image=True)
        image_pages: list[PageContent] = []
        text_pages: list[PageContent] = []
        text_top: Optional[PageContent] = None
        image_top: Optional[PageContent] = None
        if self.use_mix:
            pass
            # if self.config.r_mix_key in sample:
            #     for page in sample[self.config.r_mix_key][: self.config.top_k]:
            #         if page in sample[self.config.r_image_key]:
            #             origin_image_path = ""
            #             origin_image_path = content_list[page].image_path
            #             images.append(origin_image_path)
            #         if page in sample[self.config.r_text_key]:
            #             texts.append(content_list[page].txt.replace("\n", ""))
        else:
            if self.r_text_key in sample and sample[self.r_text_key]:
                rerank_text_list = self.page_rank(
                    sample[self.r_text_key], sample[f"{self.r_text_key}_score"]
                )
                for page in rerank_text_list[: self.top_k]:
                    text_pages.append(content_list[page])
                text_top = content_list[rerank_text_list[0]]
            if self.r_image_key in sample and sample[self.r_image_key]:
                rerank_image_list = self.page_rank(
                    sample[self.r_image_key], sample[f"{self.r_image_key}_score"]
                )
                for page in rerank_image_list[: self.top_k]:
                    image_pages.append(content_list[page])
                image_top = content_list[rerank_image_list[0]]
        # if image_top:
        #     text_pages.append(image_top)
        # if text_top:
        #     image_pages.append(text_top)
        if extra_contents:
            for content in reversed(extra_contents[: self.top_k]):
                text_pages = [page for page in text_pages if page.page != content.page]
                image_pages = [
                    page for page in image_pages if page.page != content.page
                ]
                text_pages.insert(0, content)
                image_pages.insert(0, content)
        return text_pages, image_pages

    def page_rank(self, page_numbers: list[int], scores: list[float]):
        # unique_pages: dict[int, float] = {}
        # for page, score in zip(page_numbers, scores):
        #     if page not in unique_pages or score > unique_pages[page]:
        #         unique_pages[page] = score

        # # Convert the dictionary back into a sorted list of tuples (page, score)
        # sorted_unique_pages = sorted(unique_pages.items())

        # # Step 2: Merge consecutive pages and calculate their representative score.
        # merged_groups = []
        # current_group: list[int] = []
        # current_max_score = float("-inf")

        # for page, score in sorted_unique_pages:
        #     if not current_group or page - current_group[-1] == 1:
        #         # If it's the first page or consecutive, add it to the group.
        #         current_group.append(page)
        #         current_max_score = max(current_max_score, score)
        #     else:
        #         # Otherwise, finalize the current group and start a new one.
        #         merged_groups.append((current_max_score, current_group))
        #         current_group = [page]
        #         current_max_score = score

        # # Don't forget the last group!
        # if current_group:
        #     merged_groups.append((current_max_score, current_group))

        # # Step 3: Sort groups by their representative score.
        # sorted_merged_groups = sorted(merged_groups, reverse=True)

        # # Flatten the sorted groups into a single list of pages.
        # result = [page for _, group in sorted_merged_groups for page in group]
        seen = set()
        result = []
        for num in page_numbers:
            if num not in seen:
                seen.add(num)
                result.append(num)
        return result

    def load_processed_content(
        self, sample: dict, disable_load_image=True
    ) -> list[PageContent]:
        """读取已经存储到tmp中的，经过解析的文档内容，每一页保存为content类。
        Args:
            sample (dict): _description_
            disable_load_image (bool, optional): _description_. Defaults to True.

        Returns:
            list[Content]: _description_
        """
        doc_name = self.EXTRACT_DOCUMENT_ID(sample)
        content_list = []
        for page_idx in range(self.max_page):
            im_file = self.IM_FILE(doc_name, page_idx)
            text_file = self.TEXT_FILE(doc_name, page_idx)
            if not os.path.exists(im_file):
                break
            img = None
            if not disable_load_image:
                img = self.load_image(im_file)
            txt = self.load_txt(text_file)
            content_list.append(
                PageContent(image=img, image_path=im_file, text=txt, page=page_idx + 1)
            )
        return content_list

    def count_document_page_num(self, sample: dict) -> int:
        doc_name = self.EXTRACT_DOCUMENT_ID(sample)
        count = 0
        for page_idx in range(self.max_page):
            im_file = self.IM_FILE(doc_name, page_idx)
            if not os.path.exists(im_file):
                break
            else:
                count += 1
        return count

    def load_located_contents(
        self, sample, content_names: list[str]
    ) -> tuple[list[PageContent], list[Figure]]:
        """
        通过location函数的返回，解析question中提到的Pages 和 figures
        """
        page_ids = []
        pattern = r"^(Figure|Table|Page) ([\w\u4e00-\u9fa5]+)$"
        figure_names: list[tuple[str, str]] = []
        for content in content_names:
            match = re.match(pattern, content)
            if match:
                prefix, name = match.groups()
                if prefix == "Page":
                    try:
                        page_id = int(name)
                    except ValueError as e:
                        continue
                    if page_id not in page_ids:
                        page_ids.append(page_id)
                else:
                    figure_names.append((prefix, name))
        doc_name = self.EXTRACT_DOCUMENT_ID(sample)
        page_contents: list[PageContent] = []
        if page_ids is not None:
            for page in page_ids:
                im_file = self.IM_FILE(doc_name, page - 1)
                text_file = self.TEXT_FILE(doc_name, page - 1)
                if not os.path.exists(im_file):
                    break
                txt = self.load_txt(text_file)
                img = self.load_image(im_file)
                page_contents.append(
                    PageContent(page=page, image=img, image_path=im_file, text=txt)
                )
        if not figure_names:
            return page_contents, []
        figure_list: list[Figure] = []
        try:
            with open(
                f"{self.pdffigure2_extract_path}/data/{doc_name}.json",
                "r",
                encoding="utf-8",
            ) as f:
                data = json.load(f)
                for figType, name in figure_names:
                    for item in data:
                        if item.get("name") == name and item.get("figType") == figType:
                            img_path = item["renderURL"]
                            figure_list.append(
                                Figure(
                                    page=item.get("page") + 1,
                                    image_path=img_path,
                                    text=item.get("caption", ""),
                                )
                            )
        except Exception as e:
            print(e)
        return page_contents, figure_list

    def load_image(self, file):
        pil_im = Image.open(file)
        return pil_im

    def load_txt(self, file):
        max_length = self.max_character_per_page
        with open(file, "r", encoding="utf-8") as file:
            content = file.read()
        content = content.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
        return content[:max_length]

    def extract_content(self, resolution=144):
        self.extract_figures()
        samples = self.load_data()
        for sample in tqdm(samples):
            self._extract_content(sample, resolution=resolution)

    def _extract_content(self, sample, resolution=144):
        max_pages = self.max_page
        os.makedirs(self.extract_path, exist_ok=True)
        image_list = list()
        text_list = list()
        doc_name = self.EXTRACT_DOCUMENT_ID(sample)
        with pymupdf.open(os.path.join(self.document_path, sample["doc_id"])) as pdf:
            for index, page in enumerate(pdf[:max_pages]):
                # save page as an image
                page: pymupdf.Page
                im_file = self.IM_FILE(doc_name, index)
                if not os.path.exists(im_file):
                    im = page.get_pixmap(dpi=resolution)  # type: ignore[attr-defined]
                    im.save(im_file)
                image_list.append(im_file)
                # save page text
                txt_file = self.TEXT_FILE(doc_name, index)
                if not os.path.exists(txt_file):
                    text = page.get_text("text")  # type: ignore[attr-defined]
                    with open(txt_file, "w") as f:
                        f.write(f"**Page {index + 1} **\n" + text)
                text_list.append(txt_file)

        return image_list, text_list

    def extract_figures(
        self,
    ):
        import subprocess

        try:
            data_path = os.path.join(self.pdffigure2_extract_path, "data")
            image_path = os.path.join(self.pdffigure2_extract_path, "image")
            os.makedirs(data_path, exist_ok=True)
            os.makedirs(image_path, exist_ok=True)
            result = subprocess.run(
                [
                    os.path.join(self.pdffigure2_path, "run_pdffigure2.sh"),
                    self.document_path,
                    "stat_file.json",
                    image_path + os.sep,
                    data_path + os.sep,
                    self.pdffigure2_path,
                ]
            )
        except subprocess.CalledProcessError as e:
            print(f"脚本执行失败: {e}")
        data_dir = os.path.join(self.pdffigure2_extract_path, "data")
        all_data = []

        # 遍历目录下的所有 JSON 文件
        for filename in os.listdir(data_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(data_dir, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        data_list = json.load(f)
                    except json.JSONDecodeError:
                        print(f"无法解析JSON文件：{file_path}")
                        continue

                    # 处理每个字典项
                    for item in data_list:
                        if isinstance(item, dict) and "name" in item:
                            name_value = item["name"]
                            if isinstance(name_value, str):
                                try:
                                    # 使用 roman 模块尝试转换
                                    item["name"] = str(roman.fromRoman(name_value))
                                except roman.InvalidRomanNumeralError:
                                    # 不是合法罗马数字，跳过或可记录日志
                                    pass
                        all_data.append(item)
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(data_list, f, ensure_ascii=False, indent=2)

    def load_summary(self, sample):
        doc_name = self.EXTRACT_DOCUMENT_ID(sample)
        summary_file = self.SUMMARY_FILE(doc_name)
        summary = ""
        if os.path.exists(summary_file):
            with open(summary_file, "r", encoding="utf-8") as f:
                summary = f.read()
        else:
            print(f"Summary file {summary_file} not found.")
        return summary

    def load_index_path(self, sample):
        doc_name = self.EXTRACT_DOCUMENT_ID(sample)
        index_file = self.INDEX_FILE(doc_name)
        return index_file

    def load_index(self, sample):
        summary = ""
        index_path = self.load_index_path(sample)
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                summary = f.read()
        else:
            print(f"Outline file {index_path} not found.")
        return summary


def extract_time(file_path):
    file_name = os.path.basename(file_path)
    time_str = file_name.split(".json")[0]
    return datetime.strptime(time_str, "%Y-%m-%d-%H-%M")


def find_latest_json(result_dir):
    pattern = os.path.join(result_dir, "*-*-*-*-*.json")
    files = glob.glob(pattern)
    files = [f for f in files if not f.endswith("_results.json")]
    if not files:
        print(f"Json file not found at {result_dir}")
        return None
    latest_file = max(files, key=extract_time)
    return latest_file
