import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mydatasets.BaseDataset import BaseDataset
from mydatasets.DocumentSummarizer import DocumentSummarizer
import toml  # type: ignore[import-untyped]
import argparse


def main(toml_cfg_path):
    with open(toml_cfg_path, "r") as f:
        dq_cfg = toml.load(f)
    parser = argparse.ArgumentParser(description="extract script")
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    args = parser.parse_args()
    dq_cfg["run_args"]["run_name"] = args.run_name

    dataset = BaseDataset(config=dq_cfg, dataset_name=args.dataset_name)
    dataset.extract_content()
    # docSummarizer = DocumentSummarizer(
    #    dataset=dataset, prompt=dq_cfg["prompts"]["summarize_prompt"]
    # )
    # docSummarizer.summarize_dataset()
    # docSummarizer.index_dataset(dq_cfg["prompts"]["index_document_prompt"])


if __name__ == "__main__":
    main(toml_cfg_path="/path/to/your/file.toml")
