# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocessing script before distillation.
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
    parser.add_argument("--file_path", type=str, default=None, help="The path to the data.")
    parser.add_argument("--dataset_name", type=str, default=None, help="The path to the data.")
    parser.add_argument("--cache_dir", type=str, default="tmp/", help="The path to the data.")
    parser.add_argument("--split", type=str, default="train", help="The split to parse.")
    parser.add_argument("--field_name", type=str, default="text", help="The field name of the column to parse.")
    parser.add_argument("--max_parsing_example", type=int, default=-1, help="Number of example to include.")
    parser.add_argument("--tokenizer_type", type=str, default="bert", choices=["bert", "roberta", "gpt2"])
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased", help="The tokenizer to use.")
    parser.add_argument("--dump_file", type=str, default="data/dump", help="The dump file prefix.")
    parser.add_argument("--fast_process",
                        default=False,
                        action='store_true',
                        help="Whether to use multi-processing to process the data.")
    parser.add_argument("--preprocessing_num_workers", type=int, default=10, help="Number of process to preprocess the dataset")
    
    
    
    args = parser.parse_args()
    
    # you need to at least, and only one, specify one of them.
    assert args.file_path is not None or args.dataset_name is not None
    assert (args.file_path is not None and args.dataset_name is not None) == False

    logger.info(f"Loading Tokenizer ({args.tokenizer_name})")
    if args.tokenizer_type == "bert":
        tokenizer = BertTokenizer.from_pretrained(
            args.tokenizer_name,
            cache_dir=args.cache_dir
        )
        bos = tokenizer.special_tokens_map["cls_token"]  # `[CLS]`
        sep = tokenizer.special_tokens_map["sep_token"]  # `[SEP]`
    elif args.tokenizer_type == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained(
            args.tokenizer_name,
            cache_dir=args.cache_dir
        )
        bos = tokenizer.special_tokens_map["cls_token"]  # `<s>`
        sep = tokenizer.special_tokens_map["sep_token"]  # `</s>`
    elif args.tokenizer_type == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained(
            args.tokenizer_name,
            cache_dir=args.cache_dir
        )
        bos = tokenizer.special_tokens_map["bos_token"]  # `<|endoftext|>`
        sep = tokenizer.special_tokens_map["eos_token"]  # `<|endoftext|>`

    # note that it has to be in the huggingface nature data file.
    all_datasets = []
    if args.file_path is not None:
        logger.info(f"Loading text from {args.file_path}")
        datasets = DatasetDict.load_from_disk(args.file_path)
        all_datasets += [datasets]
    elif args.dataset_name is not None:
        for dataset_n in args.dataset_name.split("+"):
            logger.info(f"Loading text from {dataset_n}")
            if dataset_n == "wikitext":
                datasets = load_dataset(
                    "wikitext", "wikitext-103-v1",
                    cache_dir=args.cache_dir
                )
            else:
                datasets = load_dataset(
                    dataset_n,
                    cache_dir=args.cache_dir
                )
            all_datasets += [datasets]
    else:
        assert False

    logger.info("Start encoding")

    rslt = []
    iter = 0
    interval = 10000
    start = time.time()
    if args.fast_process:
        logger.info(f"We will use multi-processing to process the datasets.")
        # When using line_by_line, we just tokenize each nonempty line.
        text_column_name = args.field_name
        def tokenize_function(examples):
            print(examples)
            token_ids = tokenizer.encode(
                examples[text_column_name], add_special_tokens=False
            )
            print(token_ids)
            batch_size = token_ids.shape[0]
            for i in range(batch_size):
                rslt.append(token_ids[i])
                iter += 1
                if iter % interval == 0:
                    end = time.time()
                    logger.info(f"{iter} examples processed. - {(end-start):.2f}s/{interval}expl")
                    start = time.time()
            return examples
        
        for dataset in all_datasets:
            _ = dataset[args.split].map(
                tokenize_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset line_by_line",
            )
    else:
        data = []
        for dataset in all_datasets:
            for text in dataset[args.split]:
                if args.max_parsing_example != -1:
                    if len(data) == args.max_parsing_example:
                        break
                data += [text[args.field_name]]
        logger.info(f"{len(data)} examples to process.")
        for text in data:
            text = f"{bos} {text.strip()} {sep}"
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            rslt.append(token_ids)

            iter += 1
            if iter % interval == 0:
                end = time.time()
                logger.info(f"{iter} examples processed. - {(end-start):.2f}s/{interval}expl")
                start = time.time()
    logger.info("Finished binarization")
    logger.info(f"{len(data)} examples processed.")

    dp_file = f"{args.dump_file}.{args.split}.{args.tokenizer_name}.pickle"
    vocab_size = tokenizer.vocab_size
    if vocab_size < (1 << 16):
        rslt_ = [np.uint16(d) for d in rslt]
    else:
        rslt_ = [np.int32(d) for d in rslt]
    random.shuffle(rslt_)
    logger.info(f"Dump to {dp_file}")
    with open(dp_file, "wb") as handle:
        pickle.dump(rslt_, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
