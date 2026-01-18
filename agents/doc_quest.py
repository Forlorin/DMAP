from models.base_model import BaseModel, BaseLLM
from models.remote_llm import ImageLLM, TextLLM
from mydatasets.BaseDataset import BaseDataset, PageContent, Figure
import json
from tqdm import tqdm
from typing import Any
import base64
from abc import abstractmethod
import time
from dotmap import DotMap  # type: ignore[import-untyped]
import os
import concurrent.futures
from ordered_set import OrderedSet
import bisect
from typing import Literal
from pathlib import Path


class BaseAgent:
    prompt: str
    model: BaseLLM

    def __init__(self, model: BaseLLM, prompt: str):
        self.model = model
        self.prompt = prompt

    def mm_message_format(
        self, content_type: Literal["text", "image"], content: str
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if content_type == "text":
            if "gpt" in self.model.model_name:
                result = {"type": "text", "text": content}
            elif "qwen" in self.model.model_name:
                result = {"type": "text", "text": content}
            else:
                result = {"type": "text", "text": content}
        if content_type == "image":
            if "gpt" in self.model.model_name:
                result = {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{content}"},
                }
            elif "qwen" in self.model.model_name:
                result = {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{content}"},
                }
            else:
                result = {
                    "type": "input_image",
                    "image_url": {"url": f"data:image/png;base64,{content}"},
                }

        return result

    def content2messages(
        self,
        question: str,
    ) -> list:
        """
        如果不传入page content，则将prompt和question结合进行回答，否则生成page内容
        """
        messages = [{"role": "user", "content": f"{self.prompt}{question}"}]
        return messages

    @abstractmethod
    def predict(self, question: str) -> str:
        pass


class TextAgent(BaseAgent):
    def __init__(self, prompt: str):
        self.model = TextLLM()
        self.prompt = prompt

    def content2messages(
        self,
        question: str,
        page_contents: list[PageContent] = [],
    ) -> list:
        """
        如果不传入page content，则将prompt和question结合进行回答，否则生成page内容
        """
        text = ""
        if page_contents:
            for page in page_contents:
                text += f"{page.text.replace('\n',' ')} \n"
            text = f"The following is the reference content you can use. Answer the question mentioned above based on the content below.{text}\nQuestion:{question}"
        messages = [{"role": "user", "content": f"{self.prompt}{question}\n{text}"}]
        return messages

    def predict(self, question: str, page_content=[]):
        output, _ = self.model.query(
            self.content2messages(question=question, page_contents=page_content)
        )
        return output


class ImageAgent(BaseAgent):
    def __init__(self, prompt: str):
        self.model = ImageLLM()
        self.prompt = prompt

    def content2message(
        self,
        question: str,
        page_contents: list[PageContent],
        figure_contents: list[Figure],
    ) -> list:
        message_contents: list[dict[str, Any]] = [
            self.mm_message_format(
                "text",
                f"{self.prompt}{question}",
            )
        ]
        if figure_contents:

            message_contents.append(
                self.mm_message_format(
                    "text",
                    "The next are some figures ,tables or charts that may contain the required information.\n",
                )
            )
            for figure in figure_contents:
                if not os.path.exists(figure.image_path):
                    continue
                # 读取图片并转换为base64编码
                with open(figure.image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                message_contents.extend(
                    [
                        self.mm_message_format(
                            "text",
                            f"The following is the figure from page {figure.page}. The caption of this figure is:{figure.text},the figure is:",
                        ),
                        self.mm_message_format("image", base64_image),
                    ]
                )

        if page_contents:
            message_contents.append(
                self.mm_message_format(
                    "text",
                    "The next are some document pages that may contain the required information.I will give you page images and texts\n",
                )
            )
            for page in page_contents:
                with open(page.image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                message_contents.extend(
                    [
                        self.mm_message_format(
                            "text", f"The next is the image of Page {page.page}:"
                        ),
                        self.mm_message_format("image", base64_image),
                    ]
                )
        messages = [{"role": "user", "content": message_contents}]
        return messages

    def predict(
        self,
        question: str,
        page_contents: list[PageContent] = [],
        figure_contents: list[Figure] = [],
    ) -> str:
        messages = self.content2message(
            question=question,
            page_contents=page_contents,
            figure_contents=figure_contents,
        )
        output, _ = self.model.query(messages=messages)
        return output


class GeneralAgent(BaseAgent):
    messages: list = []
    critical_prompt: str

    def __init__(self, prompt: str, critical_prompt: str):
        self.model = ImageLLM()
        self.prompt = prompt
        self.critical_prompt = critical_prompt

    def content2messages(
        self,
        question: str,
        text_page_contents: list[PageContent] = [],
        image_page_contents: list[PageContent] = [],
    ) -> list:
        message_contents: list = [
            self.mm_message_format("text", f"{self.prompt}{question}")
        ]
        if text_page_contents:
            message_contents.append(
                self.mm_message_format(
                    "text",
                    "The next are some document pages that may contain the required information.I will give you page images and texts\n",
                )
            )
            text = ""
            for page in text_page_contents:
                text += f"{page.text.replace('\n',' ')} \n"
            message_contents.append(self.mm_message_format("text", text))
        if image_page_contents:
            for page in image_page_contents:
                for text_page in text_page_contents:
                    if text_page.page == page.page:
                        continue
                with open(page.image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                message_contents.extend(
                    [
                        self.mm_message_format("text", f"Page {page.page}:"),
                        self.mm_message_format("image", base64_image),
                    ]
                )
        message = [{"role": "user", "content": message_contents}]
        return message

    def predict(
        self,
        question: str,
        text_page_contents: list[PageContent] = [],
        image_page_contents: list[PageContent] = [],
    ) -> str:
        messages = self.content2messages(
            question=question,
            text_page_contents=text_page_contents,
            image_page_contents=image_page_contents,
        )
        output, history = self.model.query(messages=messages)
        self.messages = history
        return output

    def self_reflect(self):
        answer, _ = self.model.predict(
            question=self.critical_prompt, history=self.messages
        )  # type:ignore
        self.clean_messages()
        return answer

    def clean_messages(self):
        self.messages = []


class LocateAgent(TextAgent):
    def __init__(self, prompt: str, advice_prompt: str):
        self.advice_prompt = advice_prompt
        super().__init__(prompt)

    def content2messages(
        self, question: str, page_contents: list[PageContent] = []
    ) -> list:
        return [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": question},
        ]

    def locate(self, question: str, summary: str, outline: str) -> dict[str, list[str]]:
        q = f"Question:\n{question}\nSummary:\n{summary}\nOutline:{outline}\n"
        q += f'For question :"{question}", based on the provided document summary and outline, identify the most relevant locations:'
        messages = self.content2messages(question=q)
        output, _ = self.model.query(messages=messages)
        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            start_idx = output.find("{")
            end_idx = output.rfind("}")
            if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
                print(f"can't find json {output}")
                data = {"location": ["not mentioned"]}
            potential_json = output[start_idx : end_idx + 1]
            try:
                data = json.loads(potential_json)
            except json.JSONDecodeError as e:
                print(f"can't find json: {e}")
                data = {"location": ["not mentioned"]}
        return data


class ReflectAgent(ImageAgent):
    def __init__(self, prompt: str, answer_prompt: str):
        self.answer_prompt = answer_prompt
        super().__init__(prompt)

    def reflect(self, question: str, answer: str) -> bool:
        """
        返回值：False：问题回答不完整
        返回值：True：问题回答完整
        """
        content = self.prompt.format(question=question, answer=answer)
        output = self.model.query([{"role": "user", "content": content}])
        if "no" in output:
            return False
        else:
            return True

    def get_pages(self, section_pages: list[int], page_num: int):
        if len(section_pages) <= 4:
            return section_pages
        else:
            sorted_nums = sorted(section_pages)
            pos = bisect.bisect_left(sorted_nums, page_num)
            left = pos - 1
            right = pos
            result: list[int] = []
            while len(result) < 4:
                if left >= 0 and (
                    right >= len(sorted_nums)
                    or abs(sorted_nums[left] - page_num)
                    <= abs(sorted_nums[right] - page_num)
                ):
                    result.append(sorted_nums[left])
                    left -= 1
                elif right < len(sorted_nums):
                    result.append(sorted_nums[right])
                    right += 1
                else:
                    break
            return section_pages

    def get_addition_pages(
        self,
        located_pages: list[int],
        index_data: dict[str, tuple[int, set]],
        document_length: int,
    ):
        for key, value in reversed(index_data.items()):
            if key.count(".") == 0:
                continue
            father_section = key.rpartition(".")[0]
            try:
                index_data[father_section][1].update(value[1])
            except:
                continue
        # 根据当前定位到的每一页进行扩展
        # 首先找出所有包含当前页的章节，找出其余的页
        # 如果有一页或以上，则找定位的前后页
        # 如果没有，则找当前小节的大节中的页
        # 如果都没有，则找其前后页.
        addition_pages_sub: OrderedSet[int] = OrderedSet([])
        addition_pages_father: OrderedSet[int] = OrderedSet([])
        for page_num in located_pages:
            contained_page_sections = [
                (key, value[0], value[1])
                for key, value in reversed(index_data.items())
                if page_num in value[1]
            ]
            if not contained_page_sections:
                continue
            contained_page_sections.sort(key=lambda x: x[1], reverse=True)
            # 得到当前页的最小层级
            page_section_level: int = contained_page_sections[0][1]
            for section in contained_page_sections:
                if page_section_level == section[1]:  # 在当前页所属的最小子层级中寻找
                    addition_pages_sub.update(
                        self.get_pages(list(section[2]), page_num)
                    )
                else:  # 当前层级为父层级
                    addition_pages_father.update(
                        self.get_pages(list(section[2]), page_num)
                    )
        addition_pages = addition_pages_sub.union(addition_pages_father)
        if not addition_pages:
            for page_num in located_pages:
                addition_pages.update(
                    [
                        x
                        for x in range(page_num - 1, page_num + 2)
                        if 1 <= x <= document_length
                    ]
                )
        return sorted(list(addition_pages))[0:10]

    def re_answer(self, sample: dict, dataset: BaseDataset) -> str:
        question = dataset.get_sample_question(sample)
        locations = sample.get("location")
        if not locations:
            return ""
        index_path = dataset.load_index_path(sample)
        index_data: dict[str, tuple[int, set]] = {}
        with open(index_path, "r") as f:
            for line in f:
                line = line.strip()
                try:
                    key, content = line.split(":", maxsplit=1)
                except:
                    continue
                section_level = key.count(".")
                _, pages_str = content.split("<|>")
                page_strs = pages_str.split(",")
                pages = set()
                for s in page_strs:
                    try:
                        num = int(s.strip())
                        pages.add(num)
                    except:
                        pass
                index_data[key] = (section_level, pages)
        located_pages = []
        for location in locations:
            if not (isinstance(location, str) and location.startswith("Page ")):
                continue
            located_pages.append(int(location.replace("Page ", "")))
        if not located_pages:
            located_pages = [
                sample["text-top-10-question"][0],
                sample["image-top-10-question"][0],
            ]
        pages = self.get_addition_pages(
            located_pages=located_pages,
            index_data=index_data,
            document_length=dataset.count_document_page_num(sample=sample),
        )
        page_contents, _ = dataset.load_located_contents(
            sample=sample, content_names=[f"Page {page}" for page in pages]
        )
        ans, _ = self.model.query(
            self.content2message(question=question, page_contents=page_contents)
        )
        return ans

    def content2message(
        self,
        question: str,
        page_contents: list[PageContent],
        figure_contents: list[Figure] = [],
    ) -> list:
        messages: list = [{"role": "system", "content": self.answer_prompt}]
        message_contents: list = []
        if page_contents:
            message_contents.append(
                self.mm_message_format(
                    "text",
                    f"The next are some document pages that may contain the required information.I will give you page images \n",
                ),
            )
            for page in page_contents:
                with open(page.image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                message_contents.extend(
                    [
                        self.mm_message_format(
                            "text",
                            f"The next is the image of Page {page.page}:",
                        ),
                        self.mm_message_format("image", base64_image),
                    ]
                )
        message_contents.append(self.mm_message_format("text", f"Question:{question}"))
        messages.append({"role": "user", "content": message_contents})
        return messages


class SumAgent(TextAgent):
    def sum(self, sum_question):
        ans = self.predict(question=sum_question)
        try:
            response_dict = json.loads(ans)
            answer = response_dict.get("Answer", ans)
        except:
            answer = ans
        return answer


class EvalAgent(TextAgent):
    def __init__(self, prompt: str):
        super().__init__(prompt)


class DocQuestAgents:
    general_agent: GeneralAgent
    text_agent: TextAgent
    image_agent: ImageAgent
    sum_agent: SumAgent
    locate_agent: LocateAgent

    def __init__(self, config: dict):
        prompts: dict[str, str] = config["prompts"]
        self.general_agent = GeneralAgent(
            prompts["general_agent"], critical_prompt=prompts["critical_prompt"]
        )
        self.text_agent = TextAgent(prompts["text_agent"])
        self.image_agent = ImageAgent(prompts["image_agent"])
        self.sum_agent = SumAgent(prompts["sum_agent"])
        self.locate_agent = LocateAgent(
            prompts["locate_prompt"], prompts["advice_prompt"]
        )
        self.reflect_agent = ReflectAgent(
            prompts["reflect_prompt"], prompts["reflect_answer_prompt"]
        )
        self.figure_prompt = prompts["figure_prompt"]
        self.eval_prompt = prompts["eval_prompt"]
        cfg = DotMap(config)
        self.ans_key = cfg.run_args.ans_key.format(**cfg)
        self.save_freq = cfg.run_args.save_freq
        self.gt_key = cfg.dataset.gt_key
        self.max_retry = cfg.run_args.max_retry

    def predict(
        self,
        sample: dict,
        dataset: BaseDataset,
    ):
        question = dataset.get_sample_question(sample)
        ######## 定位 ########
        # 获取summary
        if "location" in sample:
            location = sample["location"]
        else:
            summary = dataset.load_summary(sample)
            outline = dataset.load_index(sample)
            location_output = self.locate_agent.locate(
                question=question, summary=summary, outline=outline
            )
            location = location_output.get("location", [])
            try:
                assert isinstance(location, list)
            except:
                location = []
            sample["location"] = location
        locate_page_contents, locate_figure_contents = dataset.load_located_contents(
            sample, location
        )
        text_page_content, image_page_content = dataset.load_sample_retrieval_data(
            sample, extra_contents=locate_page_contents
        )
        print(
            f"Page Content Length:\n Text:{len(text_page_content)} | {[page.page for page in text_page_content]} Image:{len(image_page_content)} | {[page.page for page in image_page_content]} \n Figure:{len(locate_figure_contents)}"
        )
        print(f"### location:\n{location}")
        general_response = self.general_agent.predict(
            question=question,
            text_page_contents=text_page_content,
            image_page_contents=image_page_content,
        )
        print("### General Agent: " + general_response)
        # critical_info = self.general_agent.self_reflect()
        # print("### General Critical Agent: " + critical_info)

        # start_index = critical_info.find("{")
        # end_index = critical_info.find("}") + 1
        # critical_info = critical_info[start_index:end_index]
        text_reflection = ""
        image_reflection = ""
        # try:
        #     critical_info = json.loads(critical_info)
        #     text_reflection = critical_info.get("text", "")
        #     image_reflection = critical_info.get("image", "")
        # except Exception as e:
        # print(e)
        all_messages = "General Agent:\n" + general_response + "\n"

        # reflect_prompt = "\nYou may use the given clue:\n"
        reflect_prompt = ""
        # 第三步：并行执行 text_agent 和 image_agent
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_text = executor.submit(
                self.text_agent.predict,
                question + reflect_prompt + text_reflection,
                page_content=text_page_content,
            )
            future_image = executor.submit(
                self.image_agent.predict,
                question + reflect_prompt + image_reflection,
                page_contents=image_page_content,
                figure_contents=locate_figure_contents,
            )

            text_response = future_text.result()
            image_response = future_image.result()

        # 第四步：汇总结果
        all_messages += "Text Agent:\n" + text_response + "\n"
        all_messages += "Image Agent:\n" + image_response + "\n"

        print("### Text Agent: " + text_response)
        print("### Image Agent: " + image_response)
        final_ans = self.sum_agent.sum(all_messages)
        print(f"### Final Answer: {final_ans}")
        sample["r_general"] = general_response
        sample["r_text"] = text_response
        sample["r_image"] = image_response
        return final_ans

    def predict_dataset(self, dataset: BaseDataset, resume: bool = False):
        if resume:
            samples, _ = dataset.load_latest_results()
        else:
            samples = dataset.load_data(use_retrieval=True)

        sample_no = 0
        max_retries = self.max_retry
        for sample in tqdm(samples):
            final_ans = None
            final_messages = None
            retry_count = 0
            if resume and self.ans_key in sample:
                print(f"Skip sample {sample_no} with existing answer.")
                sample_no += 1
                continue
            while retry_count <= max_retries:
                try:
                    final_ans = self.predict(sample, dataset)
                    question = dataset.get_sample_question(sample)
                    if not self.reflect_agent.reflect(
                        question=question, answer=final_ans
                    ):
                        sample["ans_raw"] = final_ans
                        final_ans = self.reflect_agent.re_answer(sample, dataset)
                        print(
                            f"reflected\nraw:{sample.get('ans_raw', None)}\n{final_ans}"
                        )
                    break  # 成功则跳出循环
                except Exception as e:
                    retry_count += 1
                    print(
                        f"Error on sample {sample_no}, retry {retry_count}/{max_retries}: {e}"
                    )
                    if retry_count > max_retries:
                        final_ans = None  # 超出重试次数，置为 None
            time.sleep(1)
            sample[self.ans_key] = final_ans
            self.clean_messages()
            sample_no += 1
            if sample_no % self.save_freq == 0:
                path = dataset.dump_results(samples)
                print(f"Save {sample_no} results to {path}.")
        path = dataset.dump_results(samples)
        print(f"Save final results to {path}.")

    def reflect(self, raw_answer: str, sample: dict, dataset: BaseDataset) -> str:
        question = dataset.get_sample_question(sample)
        ans = raw_answer
        if not self.reflect_agent.reflect(question=question, answer=raw_answer):
            ans = self.reflect_agent.re_answer(sample, dataset)
        return ans

    def reflect_dataset(self, dataset: BaseDataset):
        samples, result_path = dataset.load_latest_results()
        sample_no = 0
        re_answered_samples: list[dict] = []
        max_retries = self.max_retry
        for sample in tqdm(samples):
            sample_no += 1
            ans = sample.get(self.ans_key, "")
            retry_count = 0
            while retry_count <= max_retries:
                try:
                    assert isinstance(ans, str)
                    question = dataset.get_sample_question(sample)
                    if not self.reflect_agent.reflect(question=question, answer=ans):
                        sample["ans_raw"] = ans
                        ans = self.reflect_agent.re_answer(sample, dataset)
                        re_answered_samples.append(sample)
                        print(f"reflected\nraw:{sample.get('ans_raw', None)}\n{ans}")
                    break  # 成功则跳出循环
                except Exception as e:
                    retry_count += 1
                    print(
                        f"Error on sample {sample_no}, retry {retry_count}/{max_retries}: {e}"
                    )
                    if retry_count > max_retries:
                        ans = None  # 超出重试次数，置为 None
            sample[self.ans_key] = ans
        r_p = dataset.dump_results(samples)
        with open(os.path.join(dataset.result_dir, "reflected_samples.json"), "w") as f:
            json.dump(re_answered_samples, f, indent=4)
        print(f"Save final results to {r_p}.")

    def clean_messages(self):
        self.general_agent.clean_messages()

    def eval(self, question: str, answer: str, ground_truth: str):
        prompt = self.eval_prompt.format(
            question=question, answer=answer, gt=ground_truth
        )
        try:
            messages = [{"role": "user", "content": f"{prompt}"}]
            generated_ans, _ = self.text_agent.model.query(messages=messages)
            result = extract_evaluation_metrics(generated_ans)
            return result
        except Exception as e:
            print(f"Error evaluating answer: {str(e)}")
            return {"binary_correctness": 0}

    def eval_dataset(self, dataset: BaseDataset):
        samples, ans_path = dataset.load_latest_results()
        samples_with_answer = []
        total_score = 0.0
        count = 0
        max_retries = self.max_retry
        try:
            for sample in tqdm(samples):
                retry_count = 0
                question = sample[dataset.question_key]
                answer = sample[self.ans_key]
                gt = sample[self.gt_key]
                if None in (question, answer, gt):
                    continue
                while retry_count <= max_retries:
                    try:
                        result = self.eval(question, answer, gt)
                        sample["binary_correctness"] = result.get(
                            "binary_correctness", 0
                        )
                        samples_with_answer.append(sample)
                        count += 1
                        total_score += sample["binary_correctness"]
                        break
                    except Exception as e:
                        time.sleep(1)
                        retry_count += 1
                        if retry_count >= max_retries:
                            count -= 1  # 如果超过最大重试次数仍未成功，则取消计数
                            break
                        print(f"Error evaluating sample: {str(e)}")
        except KeyError as e:
            print(f"{e}")
        ans_file_path_name = ans_path[:-5] + "_results.json"
        with open(ans_file_path_name, "w") as file:
            json.dump(samples_with_answer, file, indent=4)
        avg_binary_correctness = total_score / count if count > 0 else 0.0
        path = os.path.join(dataset.result_dir, "results.txt")
        with open(path, "a") as file:
            file.write("\nEvaluation Results Summary:\n")
            file.write(f"Result file: {ans_path}\n")
            file.write(f"Average Binary Correctness: {avg_binary_correctness:.3f}\n")

        print(f"Save results to {path}.")
        print(f"Average Binary Correctness: {avg_binary_correctness:.3f}\n")


def extract_evaluation_metrics(eval_str: str) -> dict[str, float | int]:
    try:
        start_index = eval_str.find("{")
        end_index = eval_str.rfind("}") + 1
        eval_str = eval_str[start_index:end_index]
        metrics = json.loads(eval_str)
        return {"binary_correctness": int(metrics.get("binary_correctness", 0))}
    except json.JSONDecodeError as e:
        return {"binary_correctness": 0}
    except Exception as e:
        return {"binary_correctness": 0}
