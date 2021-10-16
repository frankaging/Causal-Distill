"""
Another preprocessing script before distillation.
"""
import argparse
import logging
import pickle
import random
import time

import numpy as np

from transformers import BertTokenizer, GPT2Tokenizer, RobertaTokenizer

from datasets import load_dataset, load_metric
from datasets import Dataset
from datasets import DatasetDict

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess the data to avoid re-doing it several times by (tokenization + token_to_ids)."
    )
    parser.add_argument("--file_path_left", type=str, default=None, help="The path to the data.")
    parser.add_argument("--file_path_right", type=str, default=None, help="The path to the data.")
    parser.add_argument("--dump_file", type=str, default=None, help="The path to the data.")
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased", help="The tokenizer to use.")
    parser.add_argument("--dump_file", type=str, default="data/dump", help="The dump file prefix.")
    parser.add_argument("--split", type=str, default="train", help="The split to parse.")
    
    args = parser.parse_args()
    
    data_left = pickle.load(open(args.file_path_left,"rb"), protocol=pickle.HIGHEST_PROTOCOL)
    data_right = pickle.load(open(args.file_path_right,"rb"), protocol=pickle.HIGHEST_PROTOCOL)
    
    for example in data_right:
        data_left.append(example)
    
    dp_file = f"{args.dump_file}.{args.split}.{args.tokenizer_name}.pickle"
    
if __name__ == "__main__":
    main()