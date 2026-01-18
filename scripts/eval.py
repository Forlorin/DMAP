import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mydatasets.BaseDataset import BaseDataset
from agents.doc_quest import DocQuestAgents
import toml  # type: ignore[import-untyped]
import argparse


def main(toml_cfg_path):
    with open(toml_cfg_path, "r") as f:
        dq_cfg = toml.load(f)
    parser = argparse.ArgumentParser(description="predict script")
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    args = parser.parse_args()
    dq_cfg["run_args"]["run_name"] = args.run_name
    dataset = BaseDataset(dq_cfg, args.dataset_name)
    docQuestAgents = DocQuestAgents(dq_cfg)
    docQuestAgents.eval_dataset(dataset)


if __name__ == "__main__":
    main(toml_cfg_path="config/config.toml")
