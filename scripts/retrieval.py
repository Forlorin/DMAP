import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mydatasets.BaseDataset import BaseDataset
from retrieval.doc_retrieval import TextRetrieval, imageRetrieval
import toml  # type: ignore[import-untyped]
import argparse


def main(toml_cfg_path):
    with open(toml_cfg_path, "r") as f:
        dq_cfg = toml.load(f)
    os.environ["CUDA_VISIBLE_DEVICES"] = dq_cfg["retrieval"]["cuda_visible_devices"]
    parser = argparse.ArgumentParser(description="retrieval script")
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    args = parser.parse_args()
    dq_cfg["run_args"]["run_name"] = args.run_name
    dataset = BaseDataset(dq_cfg, args.dataset_name)
    dataset.sample_select_num = dq_cfg["retrieval"]["sample_select_num"]
    # image retrieval
    retrieval_model = imageRetrieval(dq_cfg)
    retrieval_model.batch_find_top_k(dataset)
    with open("/path/to/your/file.log", "w", encoding="utf-8") as f:
        f.write("OK")
    # text retrieval
    retrieval_model = TextRetrieval(dq_cfg)
    retrieval_model.find_top_k(dataset)


if __name__ == "__main__":
    main(toml_cfg_path="config/config.toml")
