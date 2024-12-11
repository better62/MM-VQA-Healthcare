import json
import os
import re
import random
import pandas as pd

from make_arrow import make_arrow, make_arrow_vqa, make_arrow_melinda, make_arrow_vqa_m3ae


def prepro_vqa_ehr_xqa():
    random.seed(42)

    data = {
        "test": []
    }

    data_root = "data/finetune_data/ehr_xqa"
    image_root = f"{data_root}"

    for split in ["test"]:
        with open(f"{data_root}/{split}set.json", "r") as fp:
            samples = json.load(fp)
            for sample in samples:
                img_path = os.path.join(image_root, sample["image_name"])
                qid = sample["qid"]
                question = sample["question"]
                answer = sample["answer"]
                answer_type = sample["answer_type"]
                data[split].append({
                    "img_path": img_path,
                    "qid": qid,
                    "question": question,
                    "answer": answer,
                    "answer_type": answer_type
                })
    make_arrow_vqa(data, "vqa_ehr_xqa", "data/finetune_arrows/")

def prepro_vqa_vqa_rad():
    random.seed(42)

    data = {
        "train": [],
        "val": [],
        "test": []
    }

    data_root = "data/finetune_data/vqa_rad/"
    image_root = f"{data_root}/images"

    for split in ["train", "val", "test"]:
        with open(f"{data_root}/{split}set.json", "r") as fp:
            samples = json.load(fp)
            for sample in samples:
                img_path = os.path.join(image_root, sample["image_name"])
                qid = sample["qid"]
                question = sample["question"]
                answer = sample["answer"]
                answer_type = sample["answer_type"]
                data[split].append({
                    "img_path": img_path,
                    "qid": qid,
                    "question": question,
                    "answer": answer,
                    "answer_type": answer_type
                })
    make_arrow_vqa(data, "vqa_vqa_rad", "data/finetune_arrows/")


def prepro_vqa_vqa_rad_m3ae():
    random.seed(42)

    data = {
        "train": [],
        "val": [],
        "test": []
    }

    data_root = "data/finetune_data/vqa_rad"
    image_root = f"{data_root}/images"

    for split in ["train", "val", "test"]:
        with open(f"{data_root}/{split}set.json", "r") as fp:
            samples = json.load(fp)
            for sample in samples:
                img_path = os.path.join(image_root, sample["image_name"])
                qid = sample["qid"]
                question = sample["question"]
                answer = sample["answer"]
                answer_type = sample["answer_type"]
                data[split].append({
                    "img_path": img_path,
                    "qid": qid,
                    "question": question,
                    "answer": answer,
                    "answer_type": answer_type
                })
    make_arrow_vqa_m3ae(data, "vqa_vqa_rad", "data/finetune_arrows_m3ae/")

if __name__ == '__main__':
    # prepro_vqa_vqa_rad() # for running M3AE with T5
    prepro_vqa_vqa_rad_m3ae() # for running M3AE (normal)
