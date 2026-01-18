import torch
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import os
import pickle
from colpali_engine.models.paligemma_colbert_architecture import ColPali  # type: ignore[import-untyped]
from colpali_engine.trainer.retrieval_evaluator import CustomEvaluator  # type: ignore[import-untyped]
from colpali_engine.utils.colpali_processing_utils import (  # type: ignore[import-untyped]
    process_images,
    process_queries,
)
from transformers import AutoProcessor
from mydatasets.BaseDataset import BaseDataset
from dotmap import DotMap  # type: ignore[import-untyped]
from ragatouille import RAGPretrainedModel  # type: ignore[import-untyped]
import json
from accelerate import Accelerator
from contextlib import redirect_stdout


class BaseRetrieval:
    def __init__(self, config):
        pass

    def prepare(self, dataset: BaseDataset):
        pass

    def find_top_k(self, dataset: BaseDataset):
        pass


class imageRetrieval(BaseRetrieval):
    def __init__(self, config):
        cfg = DotMap(config)
        model_name = "vidore/colpali"
        self.model = ColPali.from_pretrained(
            "vidore/colpaligemma-3b-mix-448-base",
            torch_dtype=torch.float32,
            device_map="auto",
        ).eval()
        self.model.load_adapter(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        acc = Accelerator()
        self.device = acc.device
        self.model = self.model.to(self.device)
        self.embed_dir = cfg.retrieval.image.embed_dir.format(**cfg)
        self.doc_key = cfg.retrieval.doc_key
        self.batch_size = cfg.retrieval.image.batch_size
        self.image_question_key = cfg.retrieval.image_question_key
        self.top_k = cfg.retrieval.top_k

        self.r_image_key = cfg.retrieval.r_image_key.format(**cfg)

    def prepare(self, dataset: BaseDataset):  # type: ignore
        os.makedirs(self.embed_dir, exist_ok=True)
        embed_path = self.embed_dir + "/" + dataset.dataset_name + "_embed.pkl"
        if os.path.exists(embed_path):
            with open(embed_path, "rb") as file:  # Use "rb" mode for binary reading
                document_embeds = pickle.load(file)
        else:
            document_embeds = {}

        samples = dataset.load_data(use_retrieval=True)
        for sample in tqdm(samples):
            if sample[self.doc_key] in document_embeds:
                continue
            content_list = dataset.load_processed_content(
                sample, disable_load_image=False
            )
            images = [content.image for content in content_list]
            dataloader = DataLoader(
                images,  # type: ignore
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=lambda x: process_images(self.processor, x).to(
                    self.model.device
                ),
            )
            image_embeds = []
            for batch_image in dataloader:
                with torch.no_grad():
                    batch_image = {
                        k: v.to(self.model.device) for k, v in batch_image.items()
                    }
                    batch_image_embed = self.model(**batch_image)
                    image_embeds.extend(batch_image_embed)
            try:
                document_embeds[sample[self.doc_key]] = torch.stack(image_embeds, dim=0)
            except:
                document_embeds[sample[self.doc_key]] = None
                print("Empty doc.")

        with open(embed_path, "wb") as f:
            pickle.dump(document_embeds, f)

        return document_embeds

    def cpu_prepare(self, dataset: BaseDataset):  # type: ignore
        os.makedirs(self.embed_dir, exist_ok=True)
        embed_path = self.embed_dir + "/" + dataset.dataset_name + "_embed.pkl"
        if os.path.exists(embed_path):
            with open(embed_path, "rb") as file:  # Use "rb" mode for binary reading
                document_embeds = pickle.load(file)
        else:
            document_embeds = {}

        samples = dataset.load_data(use_retrieval=True)
        for sample in tqdm(samples):
            if sample[self.doc_key] in document_embeds:
                continue
            content_list = dataset.load_processed_content(
                sample, disable_load_image=False
            )
            images = [content.image for content in content_list]
            dataloader = DataLoader(
                images,  # type: ignore
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=lambda x: process_images(self.processor, x).to(
                    self.model.device
                ),
            )
            image_embeds = []
            for batch_image in dataloader:
                with torch.no_grad():
                    batch_image = {
                        k: v.to(self.model.device) for k, v in batch_image.items()
                    }
                    batch_image_embed = self.model(**batch_image)
                    image_embeds.extend(batch_image_embed)
            try:
                document_embeds[sample[self.doc_key]] = torch.stack(image_embeds, dim=0)
            except:
                document_embeds[sample[self.doc_key]] = None
                print("Empty doc.")

        with open(embed_path, "wb") as f:
            pickle.dump(document_embeds, f)

        return document_embeds

    def find_sample_top_k(self, sample, document_embed, top_k: int, page_id_key: str):
        query = [sample[self.image_question_key]]
        batch_queries = process_queries(
            self.processor, query, Image.new("RGB", (448, 448), (255, 255, 255))
        ).to(self.model.device)
        with torch.no_grad():
            query_embed = self.model(**batch_queries)

        page_id_list = None
        if page_id_key in sample:
            page_id_list = sample[page_id_key]
            assert isinstance(page_id_list, list)

        retriever_evaluator = CustomEvaluator(is_multi_vector=True)
        scores = retriever_evaluator.evaluate(query_embed, document_embed)

        if page_id_list:
            scores_tensor = torch.tensor(scores)
            mask = torch.zeros_like(scores_tensor, dtype=torch.bool)
            for idx in page_id_list:
                mask[0, idx] = True
            masked_scores = torch.where(
                mask, scores_tensor, torch.full_like(scores_tensor, float("-inf"))
            )
            top_page = torch.topk(masked_scores, min(top_k, len(page_id_list)), dim=-1)
        else:
            top_page = torch.topk(
                torch.tensor(scores), min(top_k, len(scores[0])), dim=-1
            )

        top_page_scores = top_page.values.tolist()[0] if top_page is not None else []
        top_page_indices = top_page.indices.tolist()[0] if top_page is not None else []

        return top_page_indices, top_page_scores

    def find_top_k(self, dataset: BaseDataset, prepare=False):  # type: ignore
        document_embeds = self.load_document_embeds(dataset, force_prepare=True)
        top_k = self.top_k
        samples = dataset.load_data(use_retrieval=True)
        sample_no = 0
        save_freq = 20
        for sample in tqdm(samples):
            if self.r_image_key in sample:
                continue
            document_embed = document_embeds[sample[self.doc_key]]
            try:
                top_page_indices, top_page_scores = self.find_sample_top_k(
                    sample, document_embed, top_k, dataset.page_id_key
                )
            except torch.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                continue
            sample[self.r_image_key] = top_page_indices
            sample[self.r_image_key + "_score"] = top_page_scores
            sample_no += 1
            if sample_no % save_freq == 0:
                path = dataset.dump_data(samples)
        path = dataset.dump_data(samples, use_retrieval=True)
        print(f"Save retrieval results at {path}.")

    def load_document_embeds(self, dataset: BaseDataset, force_prepare=False):
        embed_path = self.embed_dir + "/" + dataset.dataset_name + "_embed.pkl"
        if os.path.exists(embed_path) and not force_prepare:
            with open(embed_path, "rb") as file:  # Use "rb" mode for binary reading
                document_embeds = pickle.load(file)
        else:
            document_embeds = self.prepare(dataset)
        return document_embeds

    def prepare_doc(self, dataset: BaseDataset, embed_path: str, sample):  # type: ignore
        os.makedirs(os.path.dirname(embed_path), exist_ok=True)
        if os.path.exists(embed_path):
            with open(embed_path, "rb") as file:  # Use "rb" mode for binary reading
                document_embed = pickle.load(file)
        content_list = dataset.load_processed_content(sample, disable_load_image=False)
        images = [content.image for content in content_list]
        dataloader = DataLoader(
            images,  # type: ignore
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=lambda x: process_images(self.processor, x).to(
                self.model.device
            ),
        )
        image_embeds = []
        for batch_image in dataloader:
            with torch.no_grad():
                batch_image = {
                    k: v.to(self.model.device) for k, v in batch_image.items()
                }
                batch_image_embed = self.model(**batch_image)
                image_embeds.extend(batch_image_embed)
        try:
            document_embed = torch.stack(image_embeds, dim=0)
        except:
            document_embed = None
            print("Empty doc.")

        with open(embed_path, "wb") as f:
            pickle.dump(document_embed, f)

        return document_embed

    def load_batch_doc_embeds(self, dataset: BaseDataset, samples):
        os.makedirs(self.embed_dir, exist_ok=True)
        embed_path = self.embed_dir + "/" + dataset.dataset_name + "_embed.pkl"
        if os.path.exists(embed_path):
            with open(embed_path, "rb") as file:  # Use "rb" mode for binary reading
                document_embeds = pickle.load(file)
        else:
            document_embeds = {}

        for sample in samples:
            if sample[self.doc_key] in document_embeds:
                continue
            content_list = dataset.load_processed_content(
                sample, disable_load_image=False
            )
            images = [content.image for content in content_list]
            dataloader = DataLoader(
                images,  # type: ignore
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=lambda x: process_images(self.processor, x).to(
                    self.model.device
                ),
            )
            image_embeds = []
            for batch_image in dataloader:
                with torch.no_grad():
                    batch_image = {
                        k: v.to(self.model.device) for k, v in batch_image.items()
                    }
                    batch_image_embed = self.model(**batch_image)
                    image_embeds.extend(batch_image_embed)
            try:
                document_embeds[sample[self.doc_key]] = torch.stack(image_embeds, dim=0)
            except:
                document_embeds[sample[self.doc_key]] = None
                print("Empty doc.")
        torch.cuda.empty_cache()
        with open(embed_path, "wb") as f:
            pickle.dump(document_embeds, f)

        return document_embeds

    def batch_find_top_k(self, dataset: BaseDataset, prepare=False):  # type: ignore
        samples = dataset.load_data(use_retrieval=True)
        top_k = self.top_k
        for sample in tqdm(samples):
            if self.r_image_key in sample:
                continue
            doc_embed_path = os.path.join(
                self.embed_dir,
                dataset.dataset_name,
                f"{sample["doc_id"].replace('.pdf','')}.pkl",
            )
            try:
                if os.path.exists(doc_embed_path):
                    with open(
                        doc_embed_path, "rb"
                    ) as file:  # Use "rb" mode for binary reading
                        document_embed = pickle.load(file)
                else:
                    with open(os.devnull, "w") as fnull:
                        with redirect_stdout(fnull):
                            document_embed = self.prepare_doc(
                                dataset, doc_embed_path, sample
                            )
                top_page_indices, top_page_scores = self.find_sample_top_k(
                    sample, document_embed, top_k, dataset.page_id_key
                )
            except torch.OutOfMemoryError as e:
                print(e)
                torch.cuda.empty_cache()
                continue
            sample[self.r_image_key] = top_page_indices
            sample[self.r_image_key + "_score"] = top_page_scores
            dataset.dump_data(samples, use_retrieval=True)
            torch.cuda.empty_cache()


class TextRetrieval(BaseRetrieval):
    def __init__(self, config):
        cfg = DotMap(config)
        self.r_text_index_key = cfg.retrieval.r_text_index_key.format(**cfg)
        self.doc_key = cfg.retrieval.doc_key
        self.text_question_key = cfg.retrieval.text_question_key
        self.top_k = cfg.retrieval.top_k
        self.r_text_key = cfg.retrieval.r_text_key.format(**cfg)

    def prepare(self, dataset: BaseDataset):
        samples = dataset.load_data(use_retrieval=True)
        RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
        doc_index: dict = {}
        error = 0
        for sample in tqdm(samples):
            if self.r_text_index_key in sample and os.path.exists(
                sample[self.r_text_index_key]
            ):
                continue
            if sample[self.doc_key] in doc_index:
                sample[self.r_text_index_key] = doc_index[sample[self.doc_key]]
                continue
            content_list = dataset.load_processed_content(sample)
            text = [content.text.replace("\n", "") for content in content_list]
            # try:
            index_path = RAG.index(
                index_name=dataset.dataset_name
                + "-"
                + self.text_question_key
                + "-"
                + sample[self.doc_key],
                collection=text,
            )
            doc_index[sample[self.doc_key]] = index_path
            sample[self.r_text_index_key] = index_path
            # except Exception as e:
            #     error += 1
            #     if error > len(samples) / 100:
            #         print("Too many error cases. Exit process.")
            #         import sys

            #         sys.exit(1)
            #     print(f"Error processing {sample[self.doc_key]}: {e}")
            #     sample[self.r_text_index_key] = ""

        dataset.dump_data(samples, use_retrieval=True)

        return samples

    def find_sample_top_k(self, sample, top_k: int, page_id_key: str):
        if not os.path.exists(sample[self.r_text_index_key] + "/pid_docid_map.json"):
            print(
                f"Index not found for {sample[self.r_text_index_key]}/pid_docid_map.json."
            )
            return [], []
        with open(sample[self.r_text_index_key] + "/pid_docid_map.json", "r") as f:
            pid_map_data = json.load(f)
        unique_values = list(dict.fromkeys(pid_map_data.values()))
        value_to_rank = {val: idx for idx, val in enumerate(unique_values)}
        pid_map = {
            int(key): value_to_rank[value] for key, value in pid_map_data.items()
        }

        query = sample[self.text_question_key]
        RAG = RAGPretrainedModel.from_index(sample[self.r_text_index_key])
        results = RAG.search(query, k=len(pid_map))

        top_page_indices = [pid_map[page["passage_id"]] for page in results]
        top_page_scores = [page["score"] for page in results]

        if page_id_key in sample:
            page_id_list = sample[page_id_key]
            assert isinstance(page_id_list, list)
            filtered_indices = []
            filtered_scores = []
            for idx, score in zip(top_page_indices, top_page_scores):
                if idx in page_id_list:
                    filtered_indices.append(idx)
                    filtered_scores.append(score)
            return filtered_indices[:top_k], filtered_scores[:top_k]

        return top_page_indices[:top_k], top_page_scores[:top_k]

    def find_top_k(self, dataset: BaseDataset, force_prepare=False):
        top_k = self.top_k
        samples = dataset.load_data(use_retrieval=True)

        if self.r_text_index_key not in samples[0] or force_prepare:
            samples = self.prepare(dataset)
        sample_no = 0
        save_freq = 20
        for sample in tqdm(samples):
            if self.r_text_key in sample:
                continue
            try:
                top_page_indices, top_page_scores = self.find_sample_top_k(
                    sample, top_k=top_k, page_id_key=dataset.page_id_key
                )
            except:
                torch.cuda.empty_cache()
                continue
            sample[self.r_text_key] = top_page_indices
            sample[self.r_text_key + "_score"] = top_page_scores
            sample_no += 1
            if sample_no % save_freq == 0:
                path = dataset.dump_data(samples)
        path = dataset.dump_data(samples, use_retrieval=True)
        print(f"Save retrieval results at {path}.")
